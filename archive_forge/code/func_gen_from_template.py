import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def gen_from_template(dir: str, template_name: str, output_name: str, replacements: List[Tuple[str, Any, int]]):
    template_path = os.path.join(dir, template_name)
    output_path = os.path.join(dir, output_name)
    with open(template_path) as f:
        content = f.read()
    for placeholder, lines, indentation in replacements:
        with open(output_path, 'w') as f:
            content = content.replace(placeholder, materialize_lines(lines, indentation))
            f.write(content)