import json
import os
import pyarrow as pa
import pyarrow.jvm as pa_jvm
import pytest
import sys
import xml.etree.ElementTree as ET
@pytest.fixture(scope='session')
def root_allocator():
    try:
        arrow_dir = os.environ['ARROW_SOURCE_DIR']
    except KeyError:
        arrow_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    pom_path = os.path.join(arrow_dir, 'java', 'pom.xml')
    tree = ET.parse(pom_path)
    version = tree.getroot().find('POM:version', namespaces={'POM': 'http://maven.apache.org/POM/4.0.0'}).text
    jar_path = os.path.join(arrow_dir, 'java', 'tools', 'target', 'arrow-tools-{}-jar-with-dependencies.jar'.format(version))
    jar_path = os.getenv('ARROW_TOOLS_JAR', jar_path)
    kwargs = {}
    kwargs['convertStrings'] = False
    jpype.startJVM(jpype.getDefaultJVMPath(), '-Djava.class.path=' + jar_path, **kwargs)
    return jpype.JPackage('org').apache.arrow.memory.RootAllocator(sys.maxsize)