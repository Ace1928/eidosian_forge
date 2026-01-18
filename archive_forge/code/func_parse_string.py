import os
import sys
import re
def parse_string(astr, env, level, line):
    lineno = '#line %d\n' % line

    def replace(match):
        name = match.group(1)
        try:
            val = env[name]
        except KeyError:
            msg = 'line %d: no definition of key "%s"' % (line, name)
            raise ValueError(msg) from None
        return val
    code = [lineno]
    struct = parse_structure(astr, level)
    if struct:
        oldend = 0
        newlevel = level + 1
        for sub in struct:
            pref = astr[oldend:sub[0]]
            head = astr[sub[0]:sub[1]]
            text = astr[sub[1]:sub[2]]
            oldend = sub[3]
            newline = line + sub[4]
            code.append(replace_re.sub(replace, pref))
            try:
                envlist = parse_loop_header(head)
            except ValueError as e:
                msg = 'line %d: %s' % (newline, e)
                raise ValueError(msg)
            for newenv in envlist:
                newenv.update(env)
                newcode = parse_string(text, newenv, newlevel, newline)
                code.extend(newcode)
        suff = astr[oldend:]
        code.append(replace_re.sub(replace, suff))
    else:
        code.append(replace_re.sub(replace, astr))
    code.append('\n')
    return ''.join(code)