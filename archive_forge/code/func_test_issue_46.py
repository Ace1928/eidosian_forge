import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_46(self):
    yaml_str = dedent("        ---\n        # Please add key/value pairs in alphabetical order\n\n        aws_s3_bucket: 'mys3bucket'\n\n        jenkins_ad_credentials:\n          bind_name: 'CN=svc-AAA-BBB-T,OU=Example,DC=COM,DC=EXAMPLE,DC=Local'\n          bind_pass: 'xxxxyyyy{'\n        ")
    d = round_trip_load(yaml_str, preserve_quotes=True)
    y = round_trip_dump(d, explicit_start=True)
    assert yaml_str == y