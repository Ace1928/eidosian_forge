import posixpath
import traceback
from os import path
from typing import Any, Dict, Generator, Iterable, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import get_full_modname, logging, status_iterator
from sphinx.util.nodes import make_refnode
def env_purge_doc(app: Sphinx, env: BuildEnvironment, docname: str) -> None:
    modules = getattr(env, '_viewcode_modules', {})
    for modname, entry in list(modules.items()):
        if entry is False:
            continue
        code, tags, used, refname = entry
        for fullname in list(used):
            if used[fullname] == docname:
                used.pop(fullname)
        if len(used) == 0:
            modules.pop(modname)