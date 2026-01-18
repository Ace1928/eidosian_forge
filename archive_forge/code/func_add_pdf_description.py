from keystoneauth1 import _utils as utils
def add_pdf_description(self):
    """Add the PDF described by links.

        The standard structure includes a link to a PDF document with the
        API specification. Add it to this entry.
        """
    self.add_link(href=self._DESC_URL + 'identity-dev-guide-2.0.pdf', rel='describedby', type='application/pdf')