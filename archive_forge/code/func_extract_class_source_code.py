from __future__ import annotations
import ast
import inspect
from abc import ABCMeta
from functools import wraps
from pathlib import Path
from jinja2 import Template
from gradio.events import EventListener
from gradio.exceptions import ComponentDefinitionError
from gradio.utils import no_raise_exception
def extract_class_source_code(code: str, class_name: str) -> tuple[str, int] | tuple[None, None]:
    class_start_line = code.find(f'class {class_name}')
    if class_start_line == -1:
        return (None, None)
    class_ast = ast.parse(code)
    for node in ast.walk(class_ast):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            segment = ast.get_source_segment(code, node)
            if not segment:
                raise ValueError('segment not found')
            return (segment, node.lineno)
    return (None, None)