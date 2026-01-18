import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def load_line(self, line, currentlevel, multikey, multibackslash):
    i = 1
    quotesplits = self._get_split_on_quotes(line)
    quoted = False
    for quotesplit in quotesplits:
        if not quoted and '=' in quotesplit:
            break
        i += quotesplit.count('=')
        quoted = not quoted
    pair = line.split('=', i)
    strictly_valid = _strictly_valid_num(pair[-1])
    if _number_with_underscores.match(pair[-1]):
        pair[-1] = pair[-1].replace('_', '')
    while len(pair[-1]) and (pair[-1][0] != ' ' and pair[-1][0] != '\t' and (pair[-1][0] != "'") and (pair[-1][0] != '"') and (pair[-1][0] != '[') and (pair[-1][0] != '{') and (pair[-1].strip() != 'true') and (pair[-1].strip() != 'false')):
        try:
            float(pair[-1])
            break
        except ValueError:
            pass
        if _load_date(pair[-1]) is not None:
            break
        if TIME_RE.match(pair[-1]):
            break
        i += 1
        prev_val = pair[-1]
        pair = line.split('=', i)
        if prev_val == pair[-1]:
            raise ValueError('Invalid date or number')
        if strictly_valid:
            strictly_valid = _strictly_valid_num(pair[-1])
    pair = ['='.join(pair[:-1]).strip(), pair[-1].strip()]
    if '.' in pair[0]:
        if '"' in pair[0] or "'" in pair[0]:
            quotesplits = self._get_split_on_quotes(pair[0])
            quoted = False
            levels = []
            for quotesplit in quotesplits:
                if quoted:
                    levels.append(quotesplit)
                else:
                    levels += [level.strip() for level in quotesplit.split('.')]
                quoted = not quoted
        else:
            levels = pair[0].split('.')
        while levels[-1] == '':
            levels = levels[:-1]
        for level in levels[:-1]:
            if level == '':
                continue
            if level not in currentlevel:
                currentlevel[level] = self.get_empty_table()
            currentlevel = currentlevel[level]
        pair[0] = levels[-1].strip()
    elif (pair[0][0] == '"' or pair[0][0] == "'") and pair[0][-1] == pair[0][0]:
        pair[0] = _unescape(pair[0][1:-1])
    k, koffset = self._load_line_multiline_str(pair[1])
    if k > -1:
        while k > -1 and pair[1][k + koffset] == '\\':
            multibackslash = not multibackslash
            k -= 1
        if multibackslash:
            multilinestr = pair[1][:-1]
        else:
            multilinestr = pair[1] + '\n'
        multikey = pair[0]
    else:
        value, vtype = self.load_value(pair[1], strictly_valid)
    try:
        currentlevel[pair[0]]
        raise ValueError('Duplicate keys!')
    except TypeError:
        raise ValueError('Duplicate keys!')
    except KeyError:
        if multikey:
            return (multikey, multilinestr, multibackslash)
        else:
            currentlevel[pair[0]] = value