from __future__ import annotations
import os
import os.path
import sys
import time
import xml.dom.minidom
from dataclasses import dataclass
from typing import Any, IO, Iterable, TYPE_CHECKING
from coverage import __version__, files
from coverage.misc import isolate_module, human_sorted, human_sorted_items
from coverage.plugin import FileReporter
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis
from coverage.types import TMorf
from coverage.version import __url__
def serialize_xml(dom: xml.dom.minidom.Document) -> str:
    """Serialize a minidom node to XML."""
    return dom.toprettyxml()