def set_all_common_headers(client, parsed_args):
    if parsed_args.all_projects is not None and isinstance(parsed_args.all_projects, bool):
        set_all_projects(client, parsed_args.all_projects)
    if hasattr(parsed_args, 'edit_managed') and parsed_args.edit_managed is not None and isinstance(parsed_args.edit_managed, bool):
        set_edit_managed(client, parsed_args.edit_managed)
    if parsed_args.sudo_project_id is not None and isinstance(parsed_args.sudo_project_id, str):
        set_sudo_project_id(client, parsed_args.sudo_project_id)
    if hasattr(parsed_args, 'hard_delete') and parsed_args.hard_delete is not None and isinstance(parsed_args.hard_delete, bool):
        set_hard_delete(client, parsed_args.hard_delete)