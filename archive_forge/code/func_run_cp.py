from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import errors_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import name_expansion
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import rm_command_util
from googlecloudsdk.command_lib.storage import stdin_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_iterator
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def run_cp(args, delete_source=False):
    """Runs implementation of cp surface with tweaks for similar commands."""
    raw_destination_url = storage_url.storage_url_from_string(args.destination)
    _validate_args(args, raw_destination_url)
    encryption_util.initialize_key_store(args)
    url_found_match_tracker = collections.OrderedDict()
    if args.include_managed_folders:
        source_expansion_iterator = _get_managed_folder_iterator(args, url_found_match_tracker)
        exit_code = _execute_copy_tasks(args=args, delete_source=False, parallelizable=False, raw_destination_url=raw_destination_url, source_expansion_iterator=source_expansion_iterator)
        if exit_code:
            return exit_code
    raw_source_string_iterator = plurality_checkable_iterator.PluralityCheckableIterator(stdin_iterator.get_urls_iterable(args.source, args.read_paths_from_stdin))
    first_source_url = storage_url.storage_url_from_string(raw_source_string_iterator.peek())
    parallelizable = _is_parallelizable(args, raw_destination_url, first_source_url)
    if args.preserve_acl:
        fields_scope = cloud_api.FieldsScope.FULL
    else:
        fields_scope = cloud_api.FieldsScope.NO_ACL
    source_expansion_iterator = name_expansion.NameExpansionIterator(raw_source_string_iterator, fields_scope=fields_scope, ignore_symlinks=args.ignore_symlinks, managed_folder_setting=folder_util.ManagedFolderSetting.DO_NOT_LIST, object_state=flags.get_object_state_from_flags(args), preserve_symlinks=args.preserve_symlinks, recursion_requested=name_expansion.RecursionSetting.YES if args.recursive else name_expansion.RecursionSetting.NO_WITH_WARNING, url_found_match_tracker=url_found_match_tracker)
    exit_code = _execute_copy_tasks(args=args, delete_source=delete_source, parallelizable=parallelizable, raw_destination_url=raw_destination_url, source_expansion_iterator=source_expansion_iterator)
    if delete_source and args.include_managed_folders:
        managed_folder_expansion_iterator = name_expansion.NameExpansionIterator(args.source, managed_folder_setting=folder_util.ManagedFolderSetting.LIST_WITHOUT_OBJECTS, raise_error_for_unmatched_urls=False, recursion_requested=name_expansion.RecursionSetting.YES, url_found_match_tracker=url_found_match_tracker)
        exit_code = rm_command_util.remove_managed_folders(args, managed_folder_expansion_iterator, task_graph_executor.multiprocessing_context.Queue())
    return exit_code