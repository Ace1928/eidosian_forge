import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_issue_130a(self):
    ys = dedent('        components:\n          server: &server_component\n            type: spark.server:ServerComponent\n            host: 0.0.0.0\n            port: 8000\n          shell: &shell_component\n            type: spark.shell:ShellComponent\n\n        services:\n          server: &server_service\n            <<: *server_component\n            port: 4000\n          shell: &shell_service\n            <<: *shell_component\n            components:\n              server: {<<: *server_service}\n        ')
    data = srsly.ruamel_yaml.safe_load(ys)
    assert data['services']['shell']['components']['server']['port'] == 4000