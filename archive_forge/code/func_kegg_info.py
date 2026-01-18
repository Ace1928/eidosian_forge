import io
from urllib.request import urlopen
import time
from Bio._utils import function_with_previous
def kegg_info(database):
    """KEGG info - Displays the current statistics of a given database.

    db - database or organism (string)

    The argument db can be a KEGG database name (e.g. 'pathway' or its
    official abbreviation, 'path'), or a KEGG organism code or T number
    (e.g. 'hsa' or 'T01001' for human).

    A valid list of organism codes and their T numbers can be obtained
    via kegg_info('organism') or https://rest.kegg.jp/list/organism

    """
    return _q('info', database)