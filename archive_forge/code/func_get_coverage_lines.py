import os
import coverage
from kivy.lang.parser import Parser
def get_coverage_lines(self):
    lines = set()
    for parser_prop in walk_parser(self):
        for line_num, line in enumerate(parser_prop.value.splitlines(), start=parser_prop.line + 1):
            if line.strip():
                lines.add(line_num)
    return lines