from __future__ import annotations
import argparse
import enum
import os
import typing as t
def register_safe_action(action_type: t.Type[argparse.Action]) -> None:
    """Register the given action as a safe action for argcomplete to use during completion if it is not already registered."""
    if argcomplete and action_type not in argcomplete.safe_actions:
        if isinstance(argcomplete.safe_actions, set):
            argcomplete.safe_actions.add(action_type)
        else:
            argcomplete.safe_actions += (action_type,)