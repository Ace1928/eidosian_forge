from langcodes.util import data_filename
def parse_registry():
    """
    Yield a sequence of dictionaries, containing the info in the included
    IANA subtag registry file.
    """
    with open(data_filename('language-subtag-registry.txt'), encoding='utf-8') as data_file:
        yield from parse_file(data_file)