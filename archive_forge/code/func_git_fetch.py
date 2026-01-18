import os, subprocess, json
def git_fetch(self, cmd='git'):
    try:
        if self.reponame is not None:
            output = run_cmd([cmd, 'remote', '-v'], cwd=os.path.dirname(self.fpath))
            repo_matches = ['/' + self.reponame + '.git', '/' + self.reponame + ' ']
            if not any((m in output for m in repo_matches)):
                return self
        output = run_cmd([cmd, 'describe', '--long', '--match', 'v*.*', '--dirty'], cwd=os.path.dirname(self.fpath))
    except Exception as e:
        if e.args[1] == 'fatal: No names found, cannot describe anything.':
            raise Exception('Cannot find any git version tags of format v*.*')
        return self
    self._update_from_vcs(output)