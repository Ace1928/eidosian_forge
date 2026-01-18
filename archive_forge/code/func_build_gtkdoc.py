from __future__ import annotations
import sys, os
import subprocess
import shutil
import argparse
from ..mesonlib import MesonException, Popen_safe, is_windows, is_cygwin, split_args
from . import destdir_join
import typing as T
def build_gtkdoc(source_root: str, build_root: str, doc_subdir: str, src_subdirs: T.List[str], main_file: str, module: str, module_version: str, html_args: T.List[str], scan_args: T.List[str], fixxref_args: T.List[str], mkdb_args: T.List[str], gobject_typesfile: str, scanobjs_args: T.List[str], run: str, ld: str, cc: str, ldflags: str, cflags: str, html_assets: T.List[str], content_files: T.List[str], ignore_headers: T.List[str], namespace: str, expand_content_files: T.List[str], mode: str, options: argparse.Namespace) -> None:
    print('Building documentation for %s' % module)
    src_dir_args = []
    for src_dir in src_subdirs:
        if not os.path.isabs(src_dir):
            dirs = [os.path.join(source_root, src_dir), os.path.join(build_root, src_dir)]
        else:
            dirs = [src_dir]
        src_dir_args += ['--source-dir=' + d for d in dirs]
    doc_src = os.path.join(source_root, doc_subdir)
    abs_out = os.path.join(build_root, doc_subdir)
    htmldir = os.path.join(abs_out, 'html')
    content_files += [main_file]
    sections = os.path.join(doc_src, module + '-sections.txt')
    if os.path.exists(sections):
        content_files.append(sections)
    overrides = os.path.join(doc_src, module + '-overrides.txt')
    if os.path.exists(overrides):
        content_files.append(overrides)
    for f in content_files:
        if not os.path.isabs(f):
            f = os.path.join(doc_src, f)
        elif os.path.commonpath([f, build_root]) == build_root:
            continue
        shutil.copyfile(f, os.path.join(abs_out, os.path.basename(f)))
    shutil.rmtree(htmldir, ignore_errors=True)
    try:
        os.mkdir(htmldir)
    except Exception:
        pass
    for f in html_assets:
        f_abs = os.path.join(doc_src, f)
        shutil.copyfile(f_abs, os.path.join(htmldir, os.path.basename(f_abs)))
    scan_cmd = [options.gtkdoc_scan, '--module=' + module] + src_dir_args
    if ignore_headers:
        scan_cmd.append('--ignore-headers=' + ' '.join(ignore_headers))
    scan_cmd += scan_args
    gtkdoc_run_check(scan_cmd, abs_out)
    if '--rebuild-types' in scan_args:
        gobject_typesfile = os.path.join(abs_out, module + '.types')
    if gobject_typesfile:
        scanobjs_cmd = [options.gtkdoc_scangobj] + scanobjs_args
        scanobjs_cmd += ['--types=' + gobject_typesfile, '--module=' + module, '--run=' + run, '--cflags=' + cflags, '--ldflags=' + ldflags, '--cc=' + cc, '--ld=' + ld, '--output-dir=' + abs_out]
        library_paths = []
        for ldflag in split_args(ldflags):
            if ldflag.startswith('-Wl,-rpath,'):
                library_paths.append(ldflag[11:])
        gtkdoc_run_check(scanobjs_cmd, build_root, library_paths)
    if mode == 'auto':
        if main_file.endswith('sgml'):
            modeflag = '--sgml-mode'
        else:
            modeflag = '--xml-mode'
    elif mode == 'xml':
        modeflag = '--xml-mode'
    elif mode == 'sgml':
        modeflag = '--sgml-mode'
    else:
        modeflag = None
    mkdb_cmd = [options.gtkdoc_mkdb, '--module=' + module, '--output-format=xml', '--expand-content-files=' + ' '.join(expand_content_files)] + src_dir_args
    if namespace:
        mkdb_cmd.append('--name-space=' + namespace)
    if modeflag:
        mkdb_cmd.append(modeflag)
    if main_file:
        mkdb_cmd.append('--main-sgml-file=' + main_file)
    mkdb_cmd += mkdb_args
    gtkdoc_run_check(mkdb_cmd, abs_out)
    mkhtml_cmd = [options.gtkdoc_mkhtml, '--path=' + os.pathsep.join((doc_src, abs_out)), module] + html_args
    if main_file:
        mkhtml_cmd.append('../' + main_file)
    else:
        mkhtml_cmd.append('%s-docs.xml' % module)
    gtkdoc_run_check(mkhtml_cmd, htmldir)
    fixref_cmd = [options.gtkdoc_fixxref, '--module=' + module, '--module-dir=html'] + fixxref_args
    gtkdoc_run_check(fixref_cmd, abs_out)
    if module_version:
        shutil.move(os.path.join(htmldir, f'{module}.devhelp2'), os.path.join(htmldir, f'{module}-{module_version}.devhelp2'))