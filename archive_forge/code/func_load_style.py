from __future__ import annotations
import json
import os
from typing import Any
from pathlib import Path
from warnings import warn
from qiskit import user_config
def load_style(style: dict | str | None) -> tuple[StyleDict, float]:
    """Utility function to load style from json files.

    Args:
        style: Depending on the type, this acts differently:

            * If a string, it can specify a supported style name (such
              as "iqp" or "clifford"). It can also specify the name of
              a custom color scheme stored as JSON file. This JSON file
              _must_ specify a complete set of colors.
            * If a dictionary, it may specify the style name via a
              ``{"name": "<desired style>"}`` entry. If this is not given,
              the default style will be used. The remaining entries in the
              dictionary can be used to override certain specs.
              E.g. ``{"name": "iqp", "ec": "#FF0000"}`` will use the ``"iqp"``
              color scheme but set the edgecolor to red.

    Returns:
        A tuple containing the style as dictionary and the default font ratio.
    """
    config = user_config.get_config()
    if style is None:
        if config:
            style = config.get('circuit_mpl_style', 'default')
        else:
            style = 'default'
    if isinstance(style, dict):
        style_name = style.get('name', 'default')
    elif isinstance(style, str):
        if style.endswith('.json'):
            style_name = style[:-5]
        else:
            style_name = style
    else:
        warn(f'Unsupported style parameter "{style}" of type {type(style)}. Will use the default style.', UserWarning, 2)
        style_name = 'default'
    if style_name in ['iqp', 'default']:
        current_style = DefaultStyle().style
    else:
        style_name = style_name + '.json'
        style_paths = []
        default_path = Path(__file__).parent / 'styles' / style_name
        style_paths.append(default_path)
        if config:
            config_path = config.get('circuit_mpl_style_path', '')
            if config_path:
                for path in config_path:
                    style_paths.append(Path(path) / style_name)
        cwd_path = Path('') / style_name
        style_paths.append(cwd_path)
        for path in style_paths:
            exp_user = path.expanduser()
            if os.path.isfile(exp_user):
                try:
                    with open(exp_user) as infile:
                        json_style = json.load(infile)
                    current_style = StyleDict(json_style)
                    break
                except json.JSONDecodeError as err:
                    warn(f"Could not decode JSON in file '{path}': {str(err)}. Will use default style.", UserWarning, 2)
                    break
                except (OSError, FileNotFoundError):
                    warn(f"Error loading JSON file '{path}'. Will use default style.", UserWarning, 2)
                    break
        else:
            warn(f"Style JSON file '{style_name}' not found in any of these locations: {', '.join(map(str, style_paths))}. Will use default style.", UserWarning, 2)
            current_style = DefaultStyle().style
    if isinstance(style, dict):
        current_style.update(style)
    def_font_ratio = 13 / 8
    return (current_style, def_font_ratio)