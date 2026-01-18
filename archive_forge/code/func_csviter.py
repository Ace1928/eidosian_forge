import csv
import logging
import re
from io import StringIO
from typing import (
from warnings import warn
from lxml import etree
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.http import Response, TextResponse
from scrapy.selector import Selector
from scrapy.utils.python import re_rsearch, to_unicode
def csviter(obj: Union[Response, str, bytes], delimiter: Optional[str]=None, headers: Optional[List[str]]=None, encoding: Optional[str]=None, quotechar: Optional[str]=None) -> Generator[Dict[str, str], Any, None]:
    """Returns an iterator of dictionaries from the given csv object

    obj can be:
    - a Response object
    - a unicode string
    - a string encoded as utf-8

    delimiter is the character used to separate fields on the given obj.

    headers is an iterable that when provided offers the keys
    for the returned dictionaries, if not the first row is used.

    quotechar is the character used to enclosure fields on the given obj.
    """
    encoding = obj.encoding if isinstance(obj, TextResponse) else encoding or 'utf-8'

    def row_to_unicode(row_: Iterable) -> List[str]:
        return [to_unicode(field, encoding) for field in row_]
    lines = StringIO(_body_or_str(obj, unicode=True))
    kwargs: Dict[str, Any] = {}
    if delimiter:
        kwargs['delimiter'] = delimiter
    if quotechar:
        kwargs['quotechar'] = quotechar
    csv_r = csv.reader(lines, **kwargs)
    if not headers:
        try:
            row = next(csv_r)
        except StopIteration:
            return
        headers = row_to_unicode(row)
    for row in csv_r:
        row = row_to_unicode(row)
        if len(row) != len(headers):
            logger.warning('ignoring row %(csvlnum)d (length: %(csvrow)d, should be: %(csvheader)d)', {'csvlnum': csv_r.line_num, 'csvrow': len(row), 'csvheader': len(headers)})
            continue
        yield dict(zip(headers, row))