import json
@property
def deserialized_content(self):
    try:
        if not self._deserialized and self.content:
            self._deserialized = json.loads(self.content)
        return self._deserialized
    except ValueError as ex:
        print('Response is not a JSON object.', ex)
    return None