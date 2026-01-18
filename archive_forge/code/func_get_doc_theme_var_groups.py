import inspect
import time
from typing import Iterable
from gradio_client.documentation import document_fn
import gradio as gr
import gradio as gr
def get_doc_theme_var_groups():
    source = inspect.getsource(gr.themes.Base.set)
    groups = []
    group, desc, variables, flat_variables = (None, None, [], [])
    for line in source.splitlines():
        line = line.strip()
        if line.startswith(')'):
            break
        elif line.startswith('# '):
            if group is not None:
                groups.append((group, desc, variables))
            group, desc = line[2:].split(': ')
            variables = []
        elif '=' in line:
            var = line.split('=')[0]
            variables.append(var)
            flat_variables.append(var)
    groups.append((group, desc, variables))
    return (groups, flat_variables)