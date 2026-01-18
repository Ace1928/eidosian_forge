import argparse
import sys
def install_completion(parser):
    preamble = dict(bash=files(__package__).joinpath('backend_complete.bash').read_text(encoding='utf-8'), zsh=files(__package__).joinpath('backend_complete.zsh').read_text(encoding='utf-8'))
    shtab.add_argument_to(parser, preamble=preamble)
    get_action(parser, '--keyring-path').complete = shtab.DIR
    get_action(parser, '--keyring-backend').complete = dict(bash='_keyring_backends', zsh='backend_complete')
    return parser