import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_encode_custom_policy_1(self):
    """
        Test base64 encoding custom policy 1 from Amazon's documentation.
        """
    expected = 'eyAKICAgIlN0YXRlbWVudCI6IFt7IAogICAgICAiUmVzb3VyY2UiOiJodHRwOi8vZDYwNDcyMWZ4YWFxeTkuY2xvdWRmcm9udC5uZXQvdHJhaW5pbmcvKiIsIAogICAgICAiQ29uZGl0aW9uIjp7IAogICAgICAgICAiSXBBZGRyZXNzIjp7IkFXUzpTb3VyY2VJcCI6IjE0NS4xNjguMTQzLjAvMjQifSwgCiAgICAgICAgICJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTI1ODIzNzIwMH0gICAgICAKICAgICAgfSAKICAgfV0gCn0K'
    encoded = self.dist._url_base64_encode(self.custom_policy_1)
    self.assertEqual(expected, encoded)