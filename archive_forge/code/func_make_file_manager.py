import argparse
import os
import pathlib
import re
from collections import Counter, defaultdict, namedtuple
from typing import Dict, List, Optional, Sequence, Set, Union
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.dest as dest
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import native_function_manager
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, context, FileManager, NamespaceHelper, Target
from torchgen.yaml_utils import YamlLoader
def make_file_manager(install_dir: str) -> FileManager:
    return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=dry_run)