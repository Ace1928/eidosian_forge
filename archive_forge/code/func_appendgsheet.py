from __future__ import absolute_import, print_function, division
from petl.util.base import Table, iterdata
from petl.compat import text_type
from petl.errors import ArgumentError as PetlArgError
def appendgsheet(table, credentials_or_client, spreadsheet, worksheet=None, open_by_key=False, include_header=False):
    """
    Append a table to an existing google shoot at either a new worksheet
    or the end of an existing worksheet.

    The `credentials_or_client` are used to authenticate with the google apis.
    For more info, check `authentication`_. 

    The `spreadsheet` is the name of the workbook to append to.

    The `worksheet` is the title of the worksheet to append to or create when it
    does not exist yet.

    Set `open_by_key` to `True` in order to treat `spreadsheet` as spreadsheet key.

    Set `include_header` to `True` if you don't want omit fieldnames as the 
    first row appended.

    .. note:: 
        The sheet index cannot be used, and None is not an option.
    """
    gspread_client = _get_gspread_client(credentials_or_client)
    wb = _open_spreadsheet(gspread_client, spreadsheet, open_by_key)
    ws = _select_worksheet(wb, worksheet, True)
    rows = table if include_header else list(iterdata(table))
    ws.append_rows(rows)
    return wb.id