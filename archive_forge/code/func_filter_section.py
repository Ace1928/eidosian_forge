import os.path
import shutil
def filter_section(lines, tag):
    filtered_lines = []
    include = True
    tag_text = '#--! %s' % tag
    for line in lines:
        if line.strip().startswith(tag_text):
            include = not include
        elif include:
            filtered_lines.append(line)
    return filtered_lines