from . import validators
from . import schema
from . import compound
from . import htmlfill
def parse_schema(form):
    """
    Given an HTML form, parse out the schema defined in it and return
    that schema.
    """
    listener = SchemaBuilder()
    p = htmlfill.FillingParser(defaults={}, listener=listener)
    p.feed(form)
    p.close()
    return listener.schema()