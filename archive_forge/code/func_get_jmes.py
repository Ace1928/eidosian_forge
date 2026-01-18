from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def get_jmes(self, jmes, *processors, re=None, **kw):
    """
        Similar to :meth:`ItemLoader.get_value` but receives a JMESPath selector
        instead of a value, which is used to extract a list of unicode strings
        from the selector associated with this :class:`ItemLoader`.

        :param jmes: the JMESPath selector to extract data from
        :type jmes: str

        :param re: a regular expression to use for extracting data from the
            selected JMESPath
        :type re: str or typing.Pattern

        Examples::

            # HTML snippet: {"name": "Color TV"}
            loader.get_jmes('name')
            # HTML snippet: {"price": the price is $1200"}
            loader.get_jmes('price', TakeFirst(), re='the price is (.*)')
        """
    values = self._get_jmesvalues(jmes)
    return self.get_value(values, *processors, re=re, **kw)