import os
import torch
import warnings
from typing import List
from string import Template
from enum import Enum
def get_safety_checker(enable_azure_content_safety, enable_sensitive_topics, enable_salesforce_content_safety, enable_llamaguard_content_safety):
    safety_checker = []
    if enable_azure_content_safety:
        safety_checker.append(AzureSaftyChecker())
    if enable_sensitive_topics:
        safety_checker.append(AuditNLGSensitiveTopics())
    if enable_salesforce_content_safety:
        safety_checker.append(SalesforceSafetyChecker())
    if enable_llamaguard_content_safety:
        safety_checker.append(LlamaGuardSafetyChecker())
    return safety_checker