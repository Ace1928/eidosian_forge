import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
def patchPandas():
    if getattr(to_html_class, 'to_html') != patched_to_html:
        setattr(to_html_class, 'to_html', patched_to_html)
    if getattr(html_formatter_class, '_write_cell') != patched_write_cell:
        setattr(html_formatter_class, '_write_cell', patched_write_cell)
    if getattr(pandas_formats.format, get_adjustment_name) != patched_get_adjustment:
        setattr(pandas_formats.format, get_adjustment_name, patched_get_adjustment)
    if orig_get_formatter and getattr(dataframeformatter_class, '_get_formatter') != patched_get_formatter:
        setattr(dataframeformatter_class, '_get_formatter', patched_get_formatter)