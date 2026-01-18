from __future__ import annotations
import json
from typing import Dict, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing_extensions import NotRequired
from speech_recognition.audio import AudioData
from speech_recognition.exceptions import RequestError, UnknownValueError
@staticmethod
def convert_to_result(response_text: str) -> Result:
    """
        >>> response_text = '''{"result":[]}
        ... {"result":[{"alternative":[{"transcript":"one two three","confidence":0.49585345},{"transcript":"1 2","confidence":0.42899391}],"final":true}],"result_index":0}
        ... '''
        >>> OutputParser.convert_to_result(response_text)
        {'alternative': [{'transcript': 'one two three', 'confidence': 0.49585345}, {'transcript': '1 2', 'confidence': 0.42899391}], 'final': True}

        >>> OutputParser.convert_to_result("")
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        >>> OutputParser.convert_to_result('\\n{"result":[]}')
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        >>> OutputParser.convert_to_result('{"result":[{"foo": "bar"}]}')
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        >>> OutputParser.convert_to_result('{"result":[{"alternative": []}]}')
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        """
    for line in response_text.split('\n'):
        if not line:
            continue
        result: list[Result] = json.loads(line)['result']
        if len(result) != 0:
            if len(result[0].get('alternative', [])) == 0:
                raise UnknownValueError()
            return result[0]
    raise UnknownValueError()