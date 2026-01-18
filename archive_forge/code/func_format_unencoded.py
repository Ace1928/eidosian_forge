import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors
def format_unencoded(self, tokensource, outfile):
    for ttype, value in tokensource:
        not_found = True
        while ttype and not_found:
            try:
                on, off = self.style_string[str(ttype)]
                spl = value.split('\n')
                for line in spl[:-1]:
                    if line:
                        outfile.write(on + line + off)
                    outfile.write('\n')
                if spl[-1]:
                    outfile.write(on + spl[-1] + off)
                not_found = False
            except KeyError:
                ttype = ttype[:-1]
        if not_found:
            outfile.write(value)