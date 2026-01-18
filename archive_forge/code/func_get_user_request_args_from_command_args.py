from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.core.util import debug_output
def get_user_request_args_from_command_args(args, metadata_type=None):
    """Returns UserRequestArgs from a command's Run method "args" parameter."""
    resource_args = None
    if metadata_type:
        if metadata_type == MetadataType.BUCKET:
            cors_file_path = _get_value_or_clear_from_flag(args, 'clear_cors', 'cors_file')
            default_encryption_key = _get_value_or_clear_from_flag(args, 'clear_default_encryption_key', 'default_encryption_key')
            default_storage_class = _get_value_or_clear_from_flag(args, 'clear_default_storage_class', 'default_storage_class')
            labels_file_path = _get_value_or_clear_from_flag(args, 'clear_labels', 'labels_file')
            lifecycle_file_path = _get_value_or_clear_from_flag(args, 'clear_lifecycle', 'lifecycle_file')
            log_bucket = _get_value_or_clear_from_flag(args, 'clear_log_bucket', 'log_bucket')
            log_object_prefix = _get_value_or_clear_from_flag(args, 'clear_log_object_prefix', 'log_object_prefix')
            retention_period = _get_value_or_clear_from_flag(args, 'clear_retention_period', 'retention_period')
            web_error_page = _get_value_or_clear_from_flag(args, 'clear_web_error_page', 'web_error_page')
            web_main_page_suffix = _get_value_or_clear_from_flag(args, 'clear_web_main_page_suffix', 'web_main_page_suffix')
            resource_args = _UserBucketArgs(acl_file_path=getattr(args, 'acl_file', None), acl_grants_to_add=getattr(args, 'add_acl_grant', None), acl_grants_to_remove=getattr(args, 'remove_acl_grant', None), autoclass_terminal_storage_class=getattr(args, 'autoclass_terminal_storage_class', None), cors_file_path=cors_file_path, default_encryption_key=default_encryption_key, default_event_based_hold=getattr(args, 'default_event_based_hold', None), default_object_acl_file_path=getattr(args, 'default_object_acl_file', None), default_object_acl_grants_to_add=getattr(args, 'add_default_object_acl_grant', None), default_object_acl_grants_to_remove=getattr(args, 'remove_default_object_acl_grant', None), default_storage_class=default_storage_class, enable_autoclass=getattr(args, 'enable_autoclass', None), enable_per_object_retention=getattr(args, 'enable_per_object_retention', None), enable_hierarchical_namespace=getattr(args, 'enable_hierarchical_namespace', None), labels_file_path=labels_file_path, labels_to_append=getattr(args, 'update_labels', None), labels_to_remove=getattr(args, 'remove_labels', None), lifecycle_file_path=lifecycle_file_path, location=getattr(args, 'location', None), log_bucket=log_bucket, log_object_prefix=log_object_prefix, placement=getattr(args, 'placement', None), public_access_prevention=_get_value_or_clear_from_flag(args, 'clear_public_access_prevention', 'public_access_prevention'), recovery_point_objective=getattr(args, 'recovery_point_objective', None), requester_pays=getattr(args, 'requester_pays', None), retention_period=retention_period, retention_period_to_be_locked=getattr(args, 'lock_retention_period', False), soft_delete_duration=_get_value_or_clear_from_flag(args, 'clear_soft_delete', 'soft_delete_duration'), uniform_bucket_level_access=getattr(args, 'uniform_bucket_level_access', None), versioning=getattr(args, 'versioning', None), web_error_page=web_error_page, web_main_page_suffix=web_main_page_suffix)
        elif metadata_type == MetadataType.OBJECT:
            cache_control = _get_value_or_clear_from_flag(args, 'clear_cache_control', 'cache_control')
            content_disposition = _get_value_or_clear_from_flag(args, 'clear_content_disposition', 'content_disposition')
            content_encoding = _get_value_or_clear_from_flag(args, 'clear_content_encoding', 'content_encoding')
            content_language = _get_value_or_clear_from_flag(args, 'clear_content_language', 'content_language')
            md5_hash = _get_value_or_clear_from_flag(args, 'clear_content_md5', 'content_md5')
            content_type = _get_value_or_clear_from_flag(args, 'clear_content_type', 'content_type')
            custom_fields_to_set = _get_value_or_clear_from_flag(args, 'clear_custom_metadata', 'custom_metadata')
            custom_time = _get_value_or_clear_from_flag(args, 'clear_custom_time', 'custom_time')
            event_based_hold = getattr(args, 'event_based_hold', None)
            preserve_acl = getattr(args, 'preserve_acl', None)
            retain_until = _get_value_or_clear_from_flag(args, 'clear_retention', 'retain_until')
            storage_class = getattr(args, 'storage_class', None)
            temporary_hold = getattr(args, 'temporary_hold', None)
            retention_mode_string = _get_value_or_clear_from_flag(args, 'clear_retention', 'retention_mode')
            if retention_mode_string in (None, CLEAR):
                retention_mode = retention_mode_string
            else:
                retention_mode = flags.RetentionMode(retention_mode_string)
            resource_args = _UserObjectArgs(acl_file_path=getattr(args, 'acl_file', None), acl_grants_to_add=getattr(args, 'add_acl_grant', None), acl_grants_to_remove=getattr(args, 'remove_acl_grant', None), cache_control=cache_control, content_disposition=content_disposition, content_encoding=content_encoding, content_language=content_language, content_type=content_type, custom_fields_to_set=custom_fields_to_set, custom_fields_to_remove=getattr(args, 'remove_custom_metadata', None), custom_fields_to_update=getattr(args, 'update_custom_metadata', None), custom_time=custom_time, event_based_hold=event_based_hold, md5_hash=md5_hash, preserve_acl=preserve_acl, retain_until=retain_until, retention_mode=retention_mode, storage_class=storage_class, temporary_hold=temporary_hold)
    gzip_settings = _get_gzip_settings_from_command_args(args)
    return _UserRequestArgs(gzip_settings=gzip_settings, manifest_path=getattr(args, 'manifest_path', None), no_clobber=getattr(args, 'no_clobber', None), override_unlocked_retention=getattr(args, 'override_unlocked_retention', None) or None, precondition_generation_match=getattr(args, 'if_generation_match', None), precondition_metageneration_match=getattr(args, 'if_metageneration_match', None), predefined_acl_string=getattr(args, 'predefined_acl', None), predefined_default_object_acl_string=getattr(args, 'predefined_default_object_acl', None), preserve_posix=getattr(args, 'preserve_posix', None), preserve_symlinks=getattr(args, 'preserve_symlinks', None), resource_args=resource_args)