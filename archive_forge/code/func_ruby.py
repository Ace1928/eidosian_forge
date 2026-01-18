import re
from ..util import unikey
from ..helpers import parse_link, parse_link_label
def ruby(md):
    """A mistune plugin to support ``<ruby>`` tag. The syntax is defined
    at https://lepture.com/en/2022/markdown-ruby-markup:

    .. code-block:: text

        [漢字(ㄏㄢˋㄗˋ)]
        [漢(ㄏㄢˋ)字(ㄗˋ)]

        [漢字(ㄏㄢˋㄗˋ)][link]
        [漢字(ㄏㄢˋㄗˋ)](/url "title")

        [link]: /url "title"

    :param md: Markdown instance
    """
    md.inline.register('ruby', RUBY_PATTERN, parse_ruby, before='link')
    if md.renderer and md.renderer.NAME == 'html':
        md.renderer.register('ruby', render_ruby)