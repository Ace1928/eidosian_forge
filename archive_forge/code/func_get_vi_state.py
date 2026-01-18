from __future__ import unicode_literals
from .defaults import load_key_bindings
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.key_binding.registry import Registry, ConditionalRegistry, MergedRegistry
def get_vi_state(self, cli):
    return cli.vi_state