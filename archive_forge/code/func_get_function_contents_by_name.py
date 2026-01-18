import os
from typing import List
def get_function_contents_by_name(lines: List[str], name: str):
    """
    Extracts a function from `lines` of segmented source code with the name `name`.

    Args:
        lines (`List[str]`):
            Source code of a script seperated by line.
        name (`str`):
            The name of the function to extract. Should be either `training_function` or `main`
    """
    if name != 'training_function' and name != 'main':
        raise ValueError(f"Incorrect function name passed: {name}, choose either 'main' or 'training_function'")
    good_lines, found_start = ([], False)
    for line in lines:
        if not found_start and f'def {name}' in line:
            found_start = True
            good_lines.append(line)
            continue
        if found_start:
            if name == 'training_function' and 'def main' in line:
                return good_lines
            if name == 'main' and 'if __name__' in line:
                return good_lines
            good_lines.append(line)