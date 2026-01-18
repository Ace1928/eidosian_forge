from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def get_html_string(self, **kwargs) -> str:
    """Return string representation of HTML formatted version of table in current
        state.

        Arguments:

        title - optional table title
        start - index of first data row to include in output
        end - index of last data row to include in output PLUS ONE (list slice style)
        fields - names of fields (columns) to include
        header - print a header showing field names (True or False)
        border - print a border around the table (True or False)
        preserve_internal_border - print a border inside the table even if
            border is disabled (True or False)
        hrules - controls printing of horizontal rules after rows.
            Allowed values: ALL, FRAME, HEADER, NONE
        vrules - controls printing of vertical rules between columns.
            Allowed values: FRAME, ALL, NONE
        int_format - controls formatting of integer data
        float_format - controls formatting of floating point data
        custom_format - controls formatting of any column using callable
        padding_width - number of spaces on either side of column data (only used if
            left and right paddings are None)
        left_padding_width - number of spaces on left hand side of column data
        right_padding_width - number of spaces on right hand side of column data
        sortby - name of field to sort rows by
        sort_key - sorting key function, applied to data points before sorting
        attributes - dictionary of name/value pairs to include as HTML attributes in the
            <table> tag
        format - Controls whether or not HTML tables are formatted to match
            styling options (True or False)
        xhtml - print <br/> tags if True, <br> tags if False"""
    options = self._get_options(kwargs)
    if options['format']:
        string = self._get_formatted_html_string(options)
    else:
        string = self._get_simple_html_string(options)
    return string