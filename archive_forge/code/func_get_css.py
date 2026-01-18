from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def get_css(self, css, *processors, re=None, **kw):
    """
        Similar to :meth:`ItemLoader.get_value` but receives a CSS selector
        instead of a value, which is used to extract a list of unicode strings
        from the selector associated with this :class:`ItemLoader`.

        :param css: the CSS selector to extract data from
        :type css: str

        :param re: a regular expression to use for extracting data from the
            selected CSS region
        :type re: str or typing.Pattern

        Examples::

            # HTML snippet: <p class="product-name">Color TV</p>
            loader.get_css('p.product-name')
            # HTML snippet: <p id="price">the price is $1200</p>
            loader.get_css('p#price', TakeFirst(), re='the price is (.*)')
        """
    values = self._get_cssvalues(css)
    return self.get_value(values, *processors, re=re, **kw)