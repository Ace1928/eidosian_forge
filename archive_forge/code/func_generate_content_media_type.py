from .draft06 import CodeGeneratorDraft06
def generate_content_media_type(self):
    """
        Means loading value when it's specified as JSON.

        .. code-block:: python

            {
                'contentMediaType': 'application/json',
            }
        """
    if self._definition['contentMediaType'] == 'application/json':
        with self.l('if isinstance({variable}, bytes):'):
            with self.l('try:'):
                self.l('{variable} = {variable}.decode("utf-8")')
            with self.l('except Exception:'):
                self.exc('{name} must encoded by utf8')
        with self.l('if isinstance({variable}, str):'):
            with self.l('try:'):
                self.l('import json')
                self.l('{variable} = json.loads({variable})')
            with self.l('except Exception:'):
                self.exc('{name} must be valid JSON')