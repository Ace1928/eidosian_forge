import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def generate_rpkg(pkg_data, rpkg_data, project_shortname, export_string, package_depends, package_imports, package_suggests, has_wildcards):
    """Generate documents for R package creation.

    Parameters
    ----------
    pkg_data
    rpkg_data
    project_shortname
    export_string
    package_depends
    package_imports
    package_suggests
    has_wildcards

    Returns
    -------
    """
    package_name = snake_case_to_camel_case(project_shortname)
    package_copyright = ''
    package_rauthors = ''
    lib_name = pkg_data.get('name')
    if rpkg_data is not None:
        if rpkg_data.get('pkg_help_title'):
            package_title = rpkg_data.get('pkg_help_title', pkg_data.get('description', ''))
        if rpkg_data.get('pkg_help_description'):
            package_description = rpkg_data.get('pkg_help_description', pkg_data.get('description', ''))
        if rpkg_data.get('pkg_copyright'):
            package_copyright = '\nCopyright: {}'.format(rpkg_data.get('pkg_copyright', ''))
    else:
        package_title = pkg_data.get('description', '')
        package_description = pkg_data.get('description', '')
    package_version = pkg_data.get('version', '0.0.1')
    if package_depends:
        package_depends = ', ' + package_depends.strip(',').lstrip()
        package_depends = re.sub('(,(?![ ]))', ', ', package_depends)
    if package_imports:
        package_imports = package_imports.strip(',').lstrip()
        package_imports = re.sub('(,(?![ ]))', ', ', package_imports)
    if package_suggests:
        package_suggests = package_suggests.strip(',').lstrip()
        package_suggests = re.sub('(,(?![ ]))', ', ', package_suggests)
    if 'bugs' in pkg_data:
        package_issues = pkg_data['bugs'].get('url', '')
    else:
        package_issues = ''
        print('Warning: a URL for bug reports was not provided. Empty string inserted.', file=sys.stderr)
    if 'homepage' in pkg_data:
        package_url = pkg_data.get('homepage', '')
    else:
        package_url = ''
        print('Warning: a homepage URL was not provided. Empty string inserted.', file=sys.stderr)
    package_author = pkg_data.get('author')
    package_author_name = package_author.split(' <')[0]
    package_author_email = package_author.split(' <')[1][:-1]
    package_author_fn = package_author_name.split(' ')[0]
    package_author_ln = package_author_name.rsplit(' ', 2)[-1]
    maintainer = pkg_data.get('maintainer', pkg_data.get('author'))
    if '<' not in package_author:
        print('Error, aborting R package generation: R packages require a properly formatted author field or installation will fail. Please include an email address enclosed within < > brackets in package.json. ', file=sys.stderr)
        sys.exit(1)
    if rpkg_data is not None:
        if rpkg_data.get('pkg_authors'):
            package_rauthors = '\nAuthors@R: {}'.format(rpkg_data.get('pkg_authors', ''))
        else:
            package_rauthors = '\nAuthors@R: person("{}", "{}", role = c("aut", "cre"), email = "{}")'.format(package_author_fn, package_author_ln, package_author_email)
    if not (os.path.isfile('LICENSE') or os.path.isfile('LICENSE.txt')):
        package_license = pkg_data.get('license', '')
    else:
        package_license = pkg_data.get('license', '') + ' + file LICENSE'
        if not os.path.isfile('LICENSE'):
            os.symlink('LICENSE.txt', 'LICENSE')
    import_string = '# AUTO GENERATED FILE - DO NOT EDIT\n\n'
    packages_string = ''
    rpackage_list = package_depends.split(', ') + package_imports.split(', ')
    rpackage_list = filter(bool, rpackage_list)
    if rpackage_list:
        for rpackage in rpackage_list:
            packages_string += '\nimport({})\n'.format(rpackage)
    if os.path.exists('vignettes'):
        vignette_builder = '\nVignetteBuilder: knitr'
        if 'knitr' not in package_suggests and 'rmarkdown' not in package_suggests:
            package_suggests += ', knitr, rmarkdown'
            package_suggests = package_suggests.lstrip(', ')
    else:
        vignette_builder = ''
    pkghelp_stub_path = os.path.join('man', package_name + '-package.Rd')
    write_js_metadata(pkg_data, project_shortname, has_wildcards)
    with open('NAMESPACE', 'w+', encoding='utf-8') as f:
        f.write(import_string)
        f.write(export_string)
        f.write(packages_string)
    with open('.Rbuildignore', 'w+', encoding='utf-8') as f2:
        f2.write(rbuild_ignore_string)
    description_string = description_template.format(package_name=package_name, package_title=package_title, package_description=package_description, package_version=package_version, package_rauthors=package_rauthors, package_depends=package_depends, package_imports=package_imports, package_suggests=package_suggests, package_license=package_license, package_copyright=package_copyright, package_url=package_url, package_issues=package_issues, vignette_builder=vignette_builder)
    with open('DESCRIPTION', 'w+', encoding='utf-8') as f3:
        f3.write(description_string)
    if rpkg_data is not None:
        if rpkg_data.get('pkg_help_description'):
            pkghelp = pkghelp_stub.format(package_name=package_name, pkg_help_title=rpkg_data.get('pkg_help_title'), pkg_help_description=rpkg_data.get('pkg_help_description'), lib_name=lib_name, maintainer=maintainer)
            with open(pkghelp_stub_path, 'w', encoding='utf-8') as f4:
                f4.write(pkghelp)