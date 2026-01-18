from __future__ import absolute_import, print_function, division
from petl.util.base import Table, iterdata
from petl.compat import text_type
from petl.errors import ArgumentError as PetlArgError
def fromgsheet(credentials_or_client, spreadsheet, worksheet=None, cell_range=None, open_by_key=False):
    """
    Extract a table from a google spreadsheet.

    The `credentials_or_client` are used to authenticate with the google apis.
    For more info, check `authentication`_. 

    The `spreadsheet` can either be the key of the spreadsheet or its name.

    The `worksheet` argument can be omitted, in which case the first
    sheet in the workbook is used by default.

    The `cell_range` argument can be used to provide a range string
    specifying the top left and bottom right corners of a set of cells to
    extract. (i.e. 'A1:C7').

    Set `open_by_key` to `True` in order to treat `spreadsheet` as spreadsheet key.

    .. note::
        - Only the top level of google drive will be searched for the 
          spreadsheet filename due to API limitations.
        - The worksheet name is case sensitive.

    Example usage follows::

        >>> from petl import fromgsheet
        >>> import gspread # doctest: +SKIP
        >>> client = gspread.service_account() # doctest: +SKIP
        >>> tbl1 = fromgsheet(client, 'example_spreadsheet', 'Sheet1') # doctest: +SKIP
        >>> tbl2 = fromgsheet(client, '9zDNETemfau0uY8ZJF0YzXEPB_5GQ75JV', credentials) # doctest: +SKIP

    This functionality relies heavily on the work by @burnash and his great 
    `gspread module`_.

    .. _gspread module: http://gspread.readthedocs.io/
    .. _authentication: http://gspread.readthedocs.io/en/latest/oauth2.html
    """
    return GoogleSheetView(credentials_or_client, spreadsheet, worksheet, cell_range, open_by_key)