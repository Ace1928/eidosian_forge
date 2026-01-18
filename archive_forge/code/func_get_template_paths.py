import mimetypes
from typing import Optional
import traitlets
from traitlets.config import Config
from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath
from nbconvert.exporters.html import HTMLExporter
from nbconvert.exporters.templateexporter import TemplateExporter
from nbconvert.filters.highlight import Highlight2HTML
from .static_file_handler import TemplateStaticFileHandler
from .utils import create_include_assets_functions
def get_template_paths(self):
    return self.template_path