def add_sudo_project_id_option(parser):
    parser.add_argument('--sudo-project-id', default=None, help='Project ID to impersonate for this command. Default: None')