import os, subprocess, json
@classmethod
def setup_version(cls, setup_path, reponame, archive_commit=None, pkgname=None, dirty='report'):
    info = {}
    git_describe = None
    pkgname = reponame if pkgname is None else pkgname
    try:
        git_describe = Version.get_setup_version(setup_path, reponame, describe=True, dirty=dirty, pkgname=pkgname, archive_commit=archive_commit)
        if git_describe is not None:
            info['git_describe'] = git_describe
    except:
        pass
    if git_describe is None:
        extracted_directory_tag = Version.extract_directory_tag(setup_path, reponame)
        if extracted_directory_tag is not None:
            info['extracted_directory_tag'] = extracted_directory_tag
        try:
            with open(os.path.join(setup_path, pkgname, '.version'), 'w') as f:
                f.write(json.dumps({'extracted_directory_tag': extracted_directory_tag}))
        except:
            print('Error in setup_version: could not write .version file.')
    info['version_string'] = Version.get_setup_version(setup_path, reponame, describe=False, dirty=dirty, pkgname=pkgname, archive_commit=archive_commit)
    try:
        with open(os.path.join(setup_path, pkgname, '.version'), 'w') as f:
            f.write(json.dumps(info))
    except:
        print('Error in setup_version: could not write .version file.')
    return info['version_string']