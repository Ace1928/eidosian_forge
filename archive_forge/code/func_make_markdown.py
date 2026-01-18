from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def make_markdown(docs, name, description, local_version, demo, space, repo, pypi_exists):
    filtered_keys = [key for key in docs if key != '__meta__']
    source = f'\n# `{name}`\n{render_version_badge(pypi_exists, local_version, name)} {render_github_badge(repo)} {render_discuss_badge(space)}\n\n{description}\n\n## Installation\n\n```bash\npip install {name}\n```\n\n## Usage\n\n```python\n{demo}\n```\n'
    docs_classes = render_class_docs_markdown(filtered_keys, docs)
    source += docs_classes
    source += render_additional_interfaces_markdown(docs['__meta__']['additional_interfaces'])
    return source