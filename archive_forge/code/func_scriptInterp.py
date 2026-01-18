import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def scriptInterp():
    sys_argv = sys.argv[:]
    usage = "PDFENCRYPT USAGE:\n\nPdfEncrypt encrypts your PDF files.\n\nLine mode usage:\n\n% pdfencrypt.exe pdffile [-o ownerpassword] | [owner ownerpassword],\n\t[-u userpassword] | [user userpassword],\n\t[-p 1|0] | [printable 1|0],\n\t[-m 1|0] | [modifiable 1|0],\n\t[-c 1|0] | [copypastable 1|0],\n\t[-a 1|0] | [annotatable 1|0],\n\t[-s savefilename] | [savefile savefilename],\n\t[-v 1|0] | [verbose 1|0],\n\t[-e128], [encrypt128],\n\t[-h] | [help]\n\n-o or owner set the owner password.\n-u or user set the user password.\n-p or printable set the printable attribute (must be 1 or 0).\n-m or modifiable sets the modifiable attribute (must be 1 or 0).\n-c or copypastable sets the copypastable attribute (must be 1 or 0).\n-a or annotatable sets the annotatable attribute (must be 1 or 0).\n-s or savefile sets the name for the output PDF file\n-v or verbose prints useful output to the screen.\n      (this defaults to 'pdffile_encrypted.pdf').\n'-e128' or 'encrypt128' allows you to use 128 bit encryption (in beta).\n'-e256' or 'encrypt256' allows you to use 256 bit encryption (in beta AES).\n\n-h or help prints this message.\n\nSee PdfEncryptIntro.pdf for more information.\n"
    known_modes = ['-o', 'owner', '-u', 'user', '-p', 'printable', '-m', 'modifiable', '-c', 'copypastable', '-a', 'annotatable', '-s', 'savefile', '-v', 'verbose', '-h', 'help', '-e128', 'encrypt128', '-e256', 'encryptAES']
    OWNER = ''
    USER = ''
    PRINTABLE = 1
    MODIFIABLE = 1
    COPYPASTABLE = 1
    ANNOTATABLE = 1
    SAVEFILE = 'encrypted.pdf'
    caller = sys_argv[0]
    argv = list(sys_argv)[1:]
    if len(argv) > 0:
        if argv[0] == '-h' or argv[0] == 'help':
            print(usage)
            return
        if len(argv) < 2:
            raise ValueError('Must include a filename and one or more arguments!')
        if argv[0] not in known_modes:
            infile = argv[0]
            argv = argv[1:]
            if not os.path.isfile(infile):
                raise ValueError("Can't open input file '%s'!" % infile)
        else:
            raise ValueError('First argument must be name of the PDF input file!')
        STRENGTH = 40
        for s, _a in ((128, ('-e128', 'encrypt128')), (256, ('-e256', 'encrypt256'))):
            for a in _a:
                if a in argv:
                    STRENGTH = s
                    argv.remove(a)
        if '-v' in argv or 'verbose' in argv:
            if '-v' in argv:
                pos = argv.index('-v')
                arg = '-v'
            elif 'verbose' in argv:
                pos = argv.index('verbose')
                arg = 'verbose'
            try:
                verbose = int(argv[pos + 1])
            except:
                verbose = 1
            argv.remove(argv[pos + 1])
            argv.remove(arg)
        else:
            from reportlab.rl_config import verbose
        arglist = (('-o', 'OWNER', OWNER, 'Owner password'), ('owner', 'OWNER', OWNER, 'Owner password'), ('-u', 'USER', USER, 'User password'), ('user', 'USER', USER, 'User password'), ('-p', 'PRINTABLE', PRINTABLE, "'Printable'"), ('printable', 'PRINTABLE', PRINTABLE, "'Printable'"), ('-m', 'MODIFIABLE', MODIFIABLE, "'Modifiable'"), ('modifiable', 'MODIFIABLE', MODIFIABLE, "'Modifiable'"), ('-c', 'COPYPASTABLE', COPYPASTABLE, "'Copypastable'"), ('copypastable', 'COPYPASTABLE', COPYPASTABLE, "'Copypastable'"), ('-a', 'ANNOTATABLE', ANNOTATABLE, "'Annotatable'"), ('annotatable', 'ANNOTATABLE', ANNOTATABLE, "'Annotatable'"), ('-s', 'SAVEFILE', SAVEFILE, 'Output file'), ('savefile', 'SAVEFILE', SAVEFILE, 'Output file'))
        binaryrequired = ('-p', 'printable', '-m', 'modifiable', 'copypastable', '-c', 'annotatable', '-a')
        for thisarg in arglist:
            if thisarg[0] in argv:
                pos = argv.index(thisarg[0])
                if thisarg[0] in binaryrequired:
                    if argv[pos + 1] not in ('1', '0'):
                        raise ValueError("%s value must be either '1' or '0'!" % thisarg[1])
                try:
                    if argv[pos + 1] not in known_modes:
                        if thisarg[0] in binaryrequired:
                            exec(thisarg[1] + ' = int(argv[pos+1])', vars())
                        else:
                            exec(thisarg[1] + ' = argv[pos+1]', vars())
                        if verbose:
                            print("%s set to: '%s'." % (thisarg[3], argv[pos + 1]))
                        argv.remove(argv[pos + 1])
                        argv.remove(thisarg[0])
                except:
                    raise 'Unable to set %s.' % thisarg[3]
        if verbose > 4:
            print('\ninfile:', infile)
            print('STRENGTH:', STRENGTH)
            print('SAVEFILE:', SAVEFILE)
            print('USER:', USER)
            print('OWNER:', OWNER)
            print('PRINTABLE:', PRINTABLE)
            print('MODIFIABLE:', MODIFIABLE)
            print('COPYPASTABLE:', COPYPASTABLE)
            print('ANNOTATABLE:', ANNOTATABLE)
            print('SAVEFILE:', SAVEFILE)
            print('VERBOSE:', verbose)
        if SAVEFILE == 'encrypted.pdf':
            if infile[-4:] == '.pdf' or infile[-4:] == '.PDF':
                tinfile = infile[:-4]
            else:
                tinfile = infile
            SAVEFILE = tinfile + '_encrypted.pdf'
        filesize = encryptPdfOnDisk(infile, SAVEFILE, USER, OWNER, PRINTABLE, MODIFIABLE, COPYPASTABLE, ANNOTATABLE, strength=STRENGTH)
        if verbose:
            print("wrote output file '%s'(%s bytes)\n  owner password is '%s'\n  user password is '%s'" % (SAVEFILE, filesize, OWNER, USER))
        if len(argv) > 0:
            raise ValueError('\nUnrecognised arguments : %s\nknown arguments are:\n%s' % (str(argv)[1:-1], known_modes))
    else:
        print(usage)