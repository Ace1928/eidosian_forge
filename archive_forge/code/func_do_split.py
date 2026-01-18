import sys
import re
def do_split(input, masked_input, search_re):
    output = []
    mask_output = []
    for line, masked_line in zip(input, masked_input):
        m = search_re.match(masked_line)
        while m:
            split = len(m.group(1))
            line = line[:split] + '\\n' + line[split:]
            masked_line = masked_line[:split] + '\\n' + masked_line[split:]
            m = search_re.match(masked_line)
        output.extend(line.split('\\n'))
        mask_output.extend(masked_line.split('\\n'))
    return (output, mask_output)