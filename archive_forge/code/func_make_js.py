from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def make_js(interfaces: dict[str, AdditionalInterface] | None=None, user_fn_refs: dict[str, list[str]] | None=None):
    """Makes the javascript code for the additional interfaces."""
    js_obj_interfaces = '{'
    if interfaces is not None:
        for interface_name, interface in interfaces.items():
            js_obj_interfaces += f'\n            {interface_name}: {interface.get('refs', None) or '[]'}, '
    js_obj_interfaces += '}'
    js_obj_user_fn_refs = '{'
    if user_fn_refs is not None:
        for class_name, refs in user_fn_refs.items():
            js_obj_user_fn_refs += f'\n          {class_name}: {refs}, '
    js_obj_user_fn_refs += '}'
    return f'function() {{\n    const refs = {js_obj_interfaces};\n    const user_fn_refs = {js_obj_user_fn_refs};\n    requestAnimationFrame(() => {{\n\n        Object.entries(user_fn_refs).forEach(([key, refs]) => {{\n            if (refs.length > 0) {{\n                const el = document.querySelector(`.${{key}}-user-fn`);\n                if (!el) return;\n                refs.forEach(ref => {{\n                    el.innerHTML = el.innerHTML.replace(\n                        new RegExp("\\\\b"+ref+"\\\\b", "g"),\n                        `<a href="#h-${{ref.toLowerCase()}}">${{ref}}</a>`\n                    );\n                }})\n            }}\n        }})\n\n        Object.entries(refs).forEach(([key, refs]) => {{\n            if (refs.length > 0) {{\n                const el = document.querySelector(`.${{key}}`);\n                if (!el) return;\n                refs.forEach(ref => {{\n                    el.innerHTML = el.innerHTML.replace(\n                        new RegExp("\\\\b"+ref+"\\\\b", "g"),\n                        `<a href="#h-${{ref.toLowerCase()}}">${{ref}}</a>`\n                    );\n                }})\n            }}\n        }})\n    }})\n}}\n'