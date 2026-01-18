import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from zipfile import ZipFile
from jinja2 import Environment, FileSystemLoader
from .. import run_command
def create_recipe(version: str, component: str, wheel_path: str, generated_files_path: Path, qt_modules: List[str]=None, local_libs: List[str]=None, plugins: List[str]=None):
    """
    Create python_for_android recipe for PySide6 and shiboken6
    """
    qt_plugins = []
    if plugins:
        for plugin in plugins:
            plugin_category, plugin_name = plugin.split('_', 1)
            qt_plugins.append((plugin_category, plugin_name))
    qt_local_libs = []
    if local_libs:
        qt_local_libs = [local_lib for local_lib in local_libs if local_lib.startswith('Qt6')]
    rcp_tmpl_path = Path(__file__).parent / 'recipes' / f'{component}'
    environment = Environment(loader=FileSystemLoader(rcp_tmpl_path))
    template = environment.get_template('__init__.tmpl.py')
    content = template.render(version=version, wheel_path=wheel_path, qt_modules=qt_modules, qt_local_libs=qt_local_libs, qt_plugins=qt_plugins)
    recipe_path = generated_files_path / 'recipes' / f'{component}'
    recipe_path.mkdir(parents=True, exist_ok=True)
    logging.info(f'[DEPLOY] Writing {component} recipe into {str(recipe_path)}')
    with open(recipe_path / '__init__.py', mode='w', encoding='utf-8') as recipe:
        recipe.write(content)