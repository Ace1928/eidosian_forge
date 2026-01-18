import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def load_inline_object(self, line, currentlevel, multikey=False, multibackslash=False):
    candidate_groups = line[1:-1].split(',')
    groups = []
    if len(candidate_groups) == 1 and (not candidate_groups[0].strip()):
        candidate_groups.pop()
    while len(candidate_groups) > 0:
        candidate_group = candidate_groups.pop(0)
        try:
            _, value = candidate_group.split('=', 1)
        except ValueError:
            raise ValueError('Invalid inline table encountered')
        value = value.strip()
        if value[0] == value[-1] and value[0] in ('"', "'") or (value[0] in '-0123456789' or value in ('true', 'false') or (value[0] == '[' and value[-1] == ']') or (value[0] == '{' and value[-1] == '}')):
            groups.append(candidate_group)
        elif len(candidate_groups) > 0:
            candidate_groups[0] = candidate_group + ',' + candidate_groups[0]
        else:
            raise ValueError('Invalid inline table value encountered')
    for group in groups:
        status = self.load_line(group, currentlevel, multikey, multibackslash)
        if status is not None:
            break