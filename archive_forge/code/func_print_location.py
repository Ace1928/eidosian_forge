import re
from typing import Optional, Tuple, cast
from .ast import Location
from .location import SourceLocation, get_location
from .source import Source
def print_location(location: Location) -> str:
    """Render a helpful description of the location in the GraphQL Source document."""
    return print_source_location(location.source, get_location(location.source, location.start))