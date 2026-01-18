from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
@message_handler
def initialize_request(self, request):
    if self._initialize_request is not None:
        raise request.isnt_valid('Session is already initialized')
    self.client_id = request('clientID', '')
    self.capabilities = self.Capabilities(self, request)
    self.expectations = self.Expectations(self, request)
    self._initialize_request = request
    exception_breakpoint_filters = [{'filter': 'raised', 'label': 'Raised Exceptions', 'default': False, 'description': 'Break whenever any exception is raised.'}, {'filter': 'uncaught', 'label': 'Uncaught Exceptions', 'default': True, 'description': 'Break when the process is exiting due to unhandled exception.'}, {'filter': 'userUnhandled', 'label': 'User Uncaught Exceptions', 'default': False, 'description': 'Break when exception escapes into library code.'}]
    return {'supportsCompletionsRequest': True, 'supportsConditionalBreakpoints': True, 'supportsConfigurationDoneRequest': True, 'supportsDebuggerProperties': True, 'supportsDelayedStackTraceLoading': True, 'supportsEvaluateForHovers': True, 'supportsExceptionInfoRequest': True, 'supportsExceptionOptions': True, 'supportsFunctionBreakpoints': True, 'supportsHitConditionalBreakpoints': True, 'supportsLogPoints': True, 'supportsModulesRequest': True, 'supportsSetExpression': True, 'supportsSetVariable': True, 'supportsValueFormattingOptions': True, 'supportsTerminateRequest': True, 'supportsGotoTargetsRequest': True, 'supportsClipboardContext': True, 'exceptionBreakpointFilters': exception_breakpoint_filters, 'supportsStepInTargetsRequest': True}