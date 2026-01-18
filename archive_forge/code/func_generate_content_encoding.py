from .draft06 import CodeGeneratorDraft06
def generate_content_encoding(self):
    """
        Means decoding value when it's encoded by base64.

        .. code-block:: python

            {
                'contentEncoding': 'base64',
            }
        """
    if self._definition['contentEncoding'] == 'base64':
        with self.l('if isinstance({variable}, str):'):
            with self.l('try:'):
                self.l('import base64')
                self.l('{variable} = base64.b64decode({variable})')
            with self.l('except Exception:'):
                self.exc('{name} must be encoded by base64')
            with self.l('if {variable} == "":'):
                self.exc('contentEncoding must be base64')