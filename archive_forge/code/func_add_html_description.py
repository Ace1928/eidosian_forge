from keystoneauth1 import _utils as utils
def add_html_description(self):
    """Add the HTML described by links.

        The standard structure includes a link to a HTML document with the
        API specification. Add it to this entry.
        """
    self.add_link(href=self._DESC_URL + 'content', rel='describedby', type='text/html')