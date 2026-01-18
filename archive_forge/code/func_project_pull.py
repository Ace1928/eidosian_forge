from pathlib import Path
from wasabi import msg
from ..util import load_project_config, logger
from .main import Arg, app
from .remote_storage import RemoteStorage, get_command_hash
from .run import update_lockfile
def project_pull(project_dir: Path, remote: str, *, verbose: bool=False):
    config = load_project_config(project_dir)
    if remote in config.get('remotes', {}):
        remote = config['remotes'][remote]
    storage = RemoteStorage(project_dir, remote)
    commands = list(config.get('commands', []))
    while commands:
        for i, cmd in enumerate(list(commands)):
            logger.debug('CMD: %s.', cmd['name'])
            deps = [project_dir / dep for dep in cmd.get('deps', [])]
            if all((dep.exists() for dep in deps)):
                cmd_hash = get_command_hash('', '', deps, cmd['script'])
                for output_path in cmd.get('outputs', []):
                    url = storage.pull(output_path, command_hash=cmd_hash)
                    logger.debug('URL: %s for %s with command hash %s', url, output_path, cmd_hash)
                    yield (url, output_path)
                out_locs = [project_dir / out for out in cmd.get('outputs', [])]
                if all((loc.exists() for loc in out_locs)):
                    update_lockfile(project_dir, cmd)
                commands.pop(i)
                break
            else:
                logger.debug('Dependency missing. Skipping %s outputs.', cmd['name'])
        else:
            break