from __future__ import absolute_import
from __future__ import print_function
import webbrowser
from ..palette import Palette
from .colorbrewer_all_schemes import COLOR_MAPS
@property
def colorbrewer2_url(self):
    """
        URL that can be used to view the color map at colorbrewer2.org.

        """
    url = 'http://colorbrewer2.org/index.html?type={0}&scheme={1}&n={2}'
    return url.format(self.type.lower(), self.name, self.number)