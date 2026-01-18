from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
def check_forms_issues() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[GenericIssueDetails]]:
    """
    Runs the form issues check for the target page. Found issues are reported
    using Audits.issueAdded event.

    :returns: 
    """
    cmd_dict: T_JSON_DICT = {'method': 'Audits.checkFormsIssues'}
    json = (yield cmd_dict)
    return [GenericIssueDetails.from_json(i) for i in json['formIssues']]